import java.util.Scanner;
public class No6 {
    public static void main(String[] args) {
        int[] arr = new int[10];
        Scanner sc = new Scanner(System.in);
        for(int i=0;i<10;i++){
            System.out.print("Read a number: ");
            arr[i] = sc.nextInt();
        }
        for(int i=9;i>=0;i--){
            System.out.println(arr[i]);
        }
    }
}

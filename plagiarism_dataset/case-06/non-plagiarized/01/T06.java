import java.util.Scanner;
public class T06 {
    public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        int[] input = new int[10];
        for(int i = 0; i < input.length; i++){
            System.out.print("Read a number: ");
            input[i] = sc.nextInt();
        }
        for(int i = 9; i >= 0; i--){
            System.out.println(input[i]);
        }
    }
}

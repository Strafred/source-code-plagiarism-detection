import java.util.Scanner;
public class Main {
    public static void main(String[] args) {
        Scanner sc=new Scanner(System.in);
        int[] numbers=new int[10];
        for(int i=0;i<10;i++){
            System.out.print("Read a number: ");
            numbers[i]=sc.nextInt();
        }
        for(int i=9;i>=0;i--){
            System.out.println(numbers[i]);
        }
    }
}

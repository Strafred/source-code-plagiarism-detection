import java.util.Scanner;
public class Level5 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in); 
        int arrAngka[] = new int[10];
        for (int x = 0; x < 10; x++) {
            System.out.print("Read a number: ");
            arrAngka[x] = sc.nextInt();
        }
        for (int x = 9; x >= 0; x--)
            System.out.println(arrAngka[x]);
    }
}
